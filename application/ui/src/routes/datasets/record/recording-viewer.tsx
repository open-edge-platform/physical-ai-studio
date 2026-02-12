import { Button, ButtonGroup, Flex, Heading, Icon, ProgressCircle, Text, ToastQueue, View } from '@geti/ui';
import { ChevronLeft } from '@geti/ui/icons';
import { useNavigate } from 'react-router';

import { SchemaTeleoperationConfig } from '../../../api/openapi-spec';
import { RobotViewer } from '../../../features/robots/controller/robot-viewer';
import { RobotModelsProvider } from '../../../features/robots/robot-models-context';
import { paths } from '../../../router';
import { CameraView } from './../camera-view';
import { useRecording } from './recording-provider';
import { useTeleoperation } from './use-teleoperation';

interface RecordingViewerProps {
    recordingConfig: SchemaTeleoperationConfig;
}

export const RecordingViewer = ({ recordingConfig }: RecordingViewerProps) => {
    const { startEpisode, saveEpisode, cancelEpisode, observation, state, disconnect } = useTeleoperation(
        recordingConfig,
        ToastQueue.negative
    );

    const navigate = useNavigate();

    const robotType = recordingConfig.environment.robots?.[0].robot.type ?? 'SO101_Follower';

    const { setIsRecording } = useRecording();
    const onStop = () => {
        disconnect();
        setIsRecording(false);
        navigate(
            paths.project.datasets.show({
                project_id: recordingConfig.dataset.project_id,
                dataset_id: recordingConfig.dataset.id!,
            })
        );
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
                <View height='size-800'>
                    <Button onPress={onStop} alignSelf={'start'}>
                        <Icon>
                            <ChevronLeft color='white' fill='white' />
                        </Icon>
                        <Text>Stop recording</Text>
                    </Button>
                </View>
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

export const RecordingPage = () => {
    const { recordingConfig } = useRecording();

    if (recordingConfig) {
        return <RecordingViewer recordingConfig={recordingConfig} />;
    }
};
