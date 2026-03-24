import { FormEvent, useEffect, useState } from 'react';

import {
    Button,
    ButtonGroup,
    ComboBox,
    Flex,
    Form,
    Heading,
    Item,
    Keyboard,
    ProgressCircle,
    StatusLight,
    Text,
} from '@geti/ui';

import { RobotViewer } from '../../../features/robots/controller/robot-viewer';
import { Observation, useRobotControl } from '../../../features/robots/robot-control-provider';
import { RobotModelsProvider } from '../../../features/robots/robot-models-context';
import { paths } from '../../../router';
import { CameraView } from './../camera-view';

import classes from './recording-viewer.module.scss';

const getActionObservationSource = (observation?: Observation): { [joint: string]: number } | undefined => {
    if (observation === undefined) {
        return undefined;
    }
    if (observation.actions !== null) {
        return observation.actions;
    }
    return observation.state;
};

export const RecordingViewer = () => {
    const { dataset, environment, observation, state, startEpisode, discardEpisode, saveEpisode, readyForRecording } =
        useRobotControl();

    if (dataset === undefined) {
        throw 'Cannot load recording viewer without dataset.';
    }
    const [task, setTask] = useState<string>(dataset.default_task);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'ArrowRight') {
                if (state.is_recording && !saveEpisode.isPending) {
                    saveEpisode.mutate();
                } else if (!state.is_recording && task !== '') {
                    startEpisode.mutate(task);
                }
            } else if (e.key === 'ArrowLeft') {
                if (state.is_recording && !saveEpisode.isPending) {
                    discardEpisode.mutate();
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [state.is_recording, saveEpisode, startEpisode, discardEpisode, task]);

    const robots = (environment.robots ?? []).map(({ robot }) => robot);

    const onStart = (e: FormEvent) => {
        e.preventDefault();
        if (task !== '') {
            startEpisode.mutate(task);
        }
    };

    if (!readyForRecording) {
        return (
            <Flex width='100%' height={'100%'} alignItems={'center'} justifyContent={'center'} direction={'column'}>
                <Heading level={2}>
                    <Text>Initializing</Text>
                    <ProgressCircle marginStart='size-200' size='S' isIndeterminate alignSelf={'center'} />
                </Heading>
                <Flex direction='column' margin='size-200'>
                    <StatusLight variant={state.dataset_loaded ? 'positive' : 'yellow'}>Dataset</StatusLight>
                    <StatusLight variant={state.environment_loaded ? 'positive' : 'yellow'}>Environment</StatusLight>
                </Flex>
                <Button
                    variant={'secondary'}
                    href={paths.project.datasets.show({
                        dataset_id: dataset.id!,
                        project_id: dataset.project_id,
                    })}
                >
                    Cancel
                </Button>
            </Flex>
        );
    }
    const observation_source = getActionObservationSource(observation.current);
    const action_values = observation_source === undefined ? undefined : Object.values(observation_source);
    const action_keys = observation_source === undefined ? undefined : Object.keys(observation_source);

    return (
        <RobotModelsProvider>
            <Flex direction={'column'} height={'100%'} position={'relative'}>
                <Form validationBehavior='native' onSubmit={onStart}>
                    <Flex justifyContent={'start'} gap='size-100' height='size-800'>
                        <ComboBox
                            isReadOnly={state.is_recording}
                            errorMessage={'A task is required in order to record.'}
                            name='Task'
                            flex
                            isRequired
                            allowsCustomValue
                            inputValue={task}
                            onInputChange={setTask}
                        >
                            <Item key={dataset.default_task}>{dataset.default_task}</Item>
                        </ComboBox>
                        {state.is_recording ? (
                            <ButtonGroup>
                                <Button
                                    isDisabled={saveEpisode.isPending}
                                    variant={'negative'}
                                    onPress={() => discardEpisode.mutate()}
                                >
                                    <Text>Discard</Text>
                                    <Keyboard UNSAFE_className={classes.hotkey}>←</Keyboard>
                                </Button>
                                <Button isPending={saveEpisode.isPending} onPress={() => saveEpisode.mutate()}>
                                    <Text>Accept</Text>
                                    <Keyboard UNSAFE_className={classes.hotkey}>→</Keyboard>
                                </Button>
                            </ButtonGroup>
                        ) : (
                            <Button type={'submit'}>
                                <Text>Start episode</Text>
                                <Keyboard UNSAFE_className={classes.hotkey}>→</Keyboard>
                            </Button>
                        )}
                    </Flex>
                </Form>

                <Flex direction={'row'} flex gap={'size-100'}>
                    <Flex direction={'column'} alignContent={'start'} flex gap={'size-30'}>
                        {environment.cameras!.map((camera) => (
                            <CameraView key={camera.id} camera={camera} observation={observation} />
                        ))}
                    </Flex>
                    <Flex flex={3} minWidth={0}>
                        <RobotViewer featureValues={action_values} featureNames={action_keys} robot={robots[0]} />
                    </Flex>
                </Flex>
            </Flex>
        </RobotModelsProvider>
    );
};
