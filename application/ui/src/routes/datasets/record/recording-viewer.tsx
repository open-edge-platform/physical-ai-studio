import { useEffect, useState } from 'react';

import { Button, ButtonGroup, Flex, Heading, Keyboard, ProgressCircle, StatusLight, Text, ToastQueue } from '@geti/ui';

import { SchemaDatasetOutput, SchemaEnvironmentWithRelations } from '../../../api/openapi-spec';
import { RobotViewer } from '../../../features/robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../../features/robots/robot-models-context';
import { Observation, useRobotControl } from '../../../features/robots/use-robot-control';
import { paths } from '../../../router';
import { CameraView } from './../camera-view';

import classes from './recording-viewer.module.scss';

interface RecordingViewerProps {
    environment: SchemaEnvironmentWithRelations;
    dataset: SchemaDatasetOutput;
}

const getActionObservationSource = (observation?: Observation): { [joint: string]: number } | undefined => {
    if (observation === undefined) {
        return undefined;
    }
    if (observation.actions !== null) {
        return observation.actions;
    }
    return observation.state;
};

export const RecordingViewer = ({ environment, dataset }: RecordingViewerProps) => {
    const { observation, state, startEpisode, discardEpisode, saveEpisode, readyForRecording } = useRobotControl({
        environment,
        dataset,
        onError: ToastQueue.negative,
    });

    const [task, _setTask] = useState<string>(dataset.default_task);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'ArrowRight') {
                if (state.is_recording && !saveEpisode.isPending) {
                    saveEpisode.mutate();
                } else if (!state.is_recording) {
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

    const backPath = paths.project.datasets.show({
        dataset_id: dataset.id!,
        project_id: dataset.project_id,
    });

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
                <Button variant={'secondary'} href={backPath}>
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
                {state.is_recording ? (
                    <ButtonGroup alignSelf='end'>
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
                    <Button onPress={() => startEpisode.mutate(task)} alignSelf={'center'}>
                        <Text>Start episode</Text>
                        <Keyboard UNSAFE_className={classes.hotkey}>→</Keyboard>
                    </Button>
                )}
            </Flex>
        </RobotModelsProvider>
    );
};
