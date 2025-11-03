import { ButtonGroup, Flex, Heading, ProgressCircle, Button, View, Well } from "@geti/ui"
import { SchemaCameraConfigOutput, SchemaEpisode, SchemaTeleoperationConfig } from "../../api/openapi-spec"
import { RobotModelsProvider } from "../../features/robots/robot-models-context"
import { RobotViewer } from "../../features/robots/controller/robot-viewer"
import { Observation, useTeleoperation } from "./record/use-teleoperation"
import { RefObject, useEffect, useRef, useState } from "react"

import classes from './episode-viewer.module.scss';
import { useRecording } from "./recording-provider"

function useInterval(callback: () => void, delay: number) {
    const savedCallback = useRef<() => void>(callback);

    useEffect(() => {
        savedCallback.current = callback;
    }, [callback]);

    useEffect(() => {
        function tick() {
            savedCallback.current();
        }
        if (delay !== null) {
            const id = setInterval(tick, delay);
            return () => clearInterval(id);
        }
    }, [delay]);
}

interface CameraViewProps {
    observation: RefObject<Observation | undefined>;
    camera: SchemaCameraConfigOutput;
}

const CameraView = ({ camera, observation }: CameraViewProps) => {
    const [img, setImg] = useState<string>();

    useInterval(() => {
        if (observation.current?.cameras[camera.name]) {
            setImg(observation.current.cameras[camera.name]);
        }
    }, 1000 / camera.fps);

    const aspectRatio = camera.width / camera.height;

    /* eslint-disable jsx-a11y/media-has-caption */
    return (
        <Flex UNSAFE_style={{ aspectRatio }}>
            <Well flex UNSAFE_style={{ position: 'relative' }}>
                <View height={'100%'} position={'relative'}>
                    {
                        img === undefined
                            ? <Flex width="100%" height="100%" justifyContent={'center'} alignItems={'center'}>
                                <ProgressCircle isIndeterminate />
                              </Flex>
                            : <img
                                alt={`Camera frame of ${camera.name}`}
                                src={`data:image/jpg;base64,${img}`}
                                style={{
                                    objectFit: 'contain',
                                    height: '100%',
                                    width: '100%',
                                }}
                            />
                    }
                </View>
                <div className={classes.cameraTag}> {camera.name} </div>
            </Well>
        </Flex>
    );
};

interface RecordingViewerProps {
    recordingConfig: SchemaTeleoperationConfig
    addEpisode: (episode: SchemaEpisode) => void
}


const formatActionDictToArray = (actions: { [key: string]: number }): number[] => {
    const jointNames = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];
    return jointNames.map((name) => actions[`${name}.pos`])
}
export const RecordingViewer = ({ recordingConfig, addEpisode }: RecordingViewerProps) => {
    const { startEpisode, saveEpisode, cancelEpisode, observation, state, disconnect } = useTeleoperation(recordingConfig, addEpisode);

    const { setIsRecording } = useRecording();
    const onStop = () => {
        disconnect();
        setIsRecording(false);
    }

    if (!state.initialized) {
        <Flex width='100%' height={'100%'} alignItems={'center'} justifyContent={'center'}>
            <Heading>Initializing</Heading>
            <ProgressCircle isIndeterminate />
        </Flex>
    }

    const actions = observation.current === undefined ? undefined : formatActionDictToArray(observation.current["actions"])

    return (
        <RobotModelsProvider>
            <Flex direction={'column'} height={'100%'} position={'relative'}>
                <Flex direction={'row'} flex gap={'size-100'}>
                    <Flex direction={'column'} alignContent={'start'} flex gap={'size-30'}>
                        {recordingConfig.cameras.map((camera) => (
                            <CameraView key={camera.id} camera={camera} observation={observation} />
                        ))}
                    </Flex>
                    <Flex flex={3} minWidth={0}>
                        <RobotViewer jointValues={actions} />
                    </Flex>
                </Flex>
                <Button onPress={onStop} alignSelf={'start'}>Stop recording</Button>
                {state.is_recording
                    ? (
                        <ButtonGroup alignSelf='end'>
                            <Button isDisabled={saveEpisode.isPending} variant={'negative'} onPress={cancelEpisode}>Discard</Button>
                            <Button isPending={saveEpisode.isPending} onPress={() => saveEpisode.mutate()}>Accept</Button>
                        </ButtonGroup>
                    )
                    :
                    <Button onPress={startEpisode} alignSelf={'center'}>Start episode</Button>
                }
            </Flex>
        </RobotModelsProvider>
    )

}
