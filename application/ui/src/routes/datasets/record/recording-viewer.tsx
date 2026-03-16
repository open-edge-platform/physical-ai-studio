import { useEffect } from 'react';

import { Button, ButtonGroup, Flex, Heading, Keyboard, ProgressCircle, StatusLight, Text, ToastQueue, View } from '@geti/ui';

import { SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { RobotViewer } from '../../../features/robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../../features/robots/robot-models-context';
import { paths } from '../../../router';
import { CameraView } from './../camera-view';

import classes from './recording-viewer.module.scss';
import { useInference } from '../../models/inference/use-inference';

interface RecordingViewerProps {
    recordingConfig: SchemaTeleoperationConfig;
}

export const RecordingViewer = ({ recordingConfig }: RecordingViewerProps) => {
    const { observation, state, startEpisode, discardEpisode, saveEpisode, readyForRecording } = useInference(
        {
            environment: recordingConfig.environment,
            dataset: recordingConfig.dataset,
            onError: ToastQueue.negative
        }
    );

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'ArrowRight') {
                if (state.is_recording && !saveEpisode.isPending) {
                    saveEpisode.mutate();
                } else if (!state.is_recording) {
                    startEpisode.mutate(recordingConfig.task);
                }
            } else if (e.key === 'ArrowLeft') {
                if (state.is_recording && !saveEpisode.isPending) {
                    discardEpisode.mutate()
                }
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [state.is_recording, saveEpisode, startEpisode, discardEpisode]);

    const robots = (recordingConfig.environment.robots ?? []).map(({ robot }) => robot);

    const backPath = paths.project.datasets.show({
        dataset_id: recordingConfig.dataset.id!,
        project_id: recordingConfig.dataset.project_id,
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

    const action_values = observation.current === undefined ? undefined : Object.values(observation.current['state']); // TODO: Use actions?
    const action_keys = observation.current === undefined ? undefined : Object.keys(observation.current['state']);

    return (
        <RobotModelsProvider>
            <Flex direction={'column'} height={'100%'} position={'relative'}>
                <Flex direction={'row'} flex gap={'size-100'}>
                    <Flex direction={'column'} alignContent={'start'} flex gap={'size-30'}>
                        {recordingConfig.environment.cameras!.map((camera) => (
                            <CameraView key={camera.id} camera={camera} observation={observation} />
                        ))}
                    </Flex>
                    <Flex flex={3} minWidth={0}>
                        <RobotViewer featureValues={action_values} featureNames={action_keys} robot={robots[0]} />
                    </Flex>
                </Flex>
                {state.is_recording ? (
                    <ButtonGroup alignSelf='end'>
                        <Button isDisabled={saveEpisode.isPending} variant={'negative'} onPress={() => discardEpisode.mutate()}>
                            <Text>Discard</Text>
                            <Keyboard UNSAFE_className={classes.hotkey}>←</Keyboard>
                        </Button>
                        <Button isPending={saveEpisode.isPending} onPress={() => saveEpisode.mutate()}>
                            <Text>Accept</Text>
                            <Keyboard UNSAFE_className={classes.hotkey}>→</Keyboard>
                        </Button>
                    </ButtonGroup>
                ) : (
                    <Button onPress={() => startEpisode.mutate(recordingConfig.task)} alignSelf={'center'}>
                        <Text>Start episode</Text>
                        <Keyboard UNSAFE_className={classes.hotkey}>→</Keyboard>
                    </Button>
                )}
            </Flex>
        </RobotModelsProvider>
    );
};
