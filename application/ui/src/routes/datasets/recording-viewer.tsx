import { Button, ButtonGroup, Flex, Heading, ProgressCircle, ToastQueue } from '@geti/ui';

import { SchemaEpisode, SchemaTeleoperationConfig } from '../../api/openapi-spec';
import { RobotViewer } from '../../features/robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../features/robots/robot-models-context';
import { CameraView } from './camera-view';
import { useTeleoperation } from './record/use-teleoperation';
import { useRecording } from './recording-provider';

interface RecordingViewerProps {
    recordingConfig: SchemaTeleoperationConfig;
    addEpisode: (episode: SchemaEpisode) => void;
}

export const RecordingViewer = ({ recordingConfig, addEpisode }: RecordingViewerProps) => {
    const { startEpisode, saveEpisode, cancelEpisode, observation, state, disconnect } = useTeleoperation(
        recordingConfig,
        addEpisode,
        ToastQueue.negative
    );

    const robotType = recordingConfig.environment.robots?.[0].robot.type ?? 'SO101_Follower';

    const { setIsRecording } = useRecording();
    const onStop = () => {
        disconnect();
        setIsRecording(false);
    };

    if (!state.initialized) {
        return (
            <Flex
                width='100%'
                height={'100%'}
                direction='column'
                gap={'size-100'}
                alignItems={'center'}
                justifyContent={'center'}
            >
                <Heading>Initializing</Heading>
                <ProgressCircle isIndeterminate />

                <Button onPress={onStop}>Cancel</Button>
            </Flex>
        );
    }

    const action_values = observation.current === undefined ? undefined : Object.values(observation.current['actions']);
    const action_keys = observation.current === undefined ? undefined : Object.keys(observation.current['actions']);

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
                        <RobotViewer featureValues={action_values} featureNames={action_keys} robotType={robotType} />
                    </Flex>
                </Flex>
                <Button onPress={onStop} alignSelf={'start'}>
                    Stop recording
                </Button>
                {state.is_recording ? (
                    <ButtonGroup alignSelf='end'>
                        <Button isDisabled={saveEpisode.isPending} variant={'negative'} onPress={cancelEpisode}>
                            Discard
                        </Button>
                        <Button isPending={saveEpisode.isPending} onPress={() => saveEpisode.mutate()}>
                            Accept
                        </Button>
                    </ButtonGroup>
                ) : (
                    <Button onPress={startEpisode} alignSelf={'center'}>
                        Start episode
                    </Button>
                )}
            </Flex>
        </RobotModelsProvider>
    );
};
