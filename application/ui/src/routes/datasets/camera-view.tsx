import { Well, View, Flex, ProgressCircle } from "@geti/ui"

import { RefObject, useState } from "react";
import { Observation } from "./record/use-teleoperation";
import { SchemaCameraConfigOutput } from "../../api/openapi-spec";

import classes from './episode-viewer.module.scss';
import { useInterval } from "./use-interval";


interface CameraViewProps {
    observation: RefObject<Observation | undefined>;
    camera: SchemaCameraConfigOutput;
}

export const CameraView = ({ camera, observation }: CameraViewProps) => {
    const [img, setImg] = useState<string>();

    useInterval(() => {
        if (observation.current?.cameras[camera.name]) {
            setImg(observation.current.cameras[camera.name]);
        }
    }, 1000 / camera.fps);

    const aspectRatio = camera.width / camera.height;

    return (
        <Flex UNSAFE_style={{ aspectRatio }}>
            <Well flex UNSAFE_style={{ position: 'relative' }}>
                <View height={'100%'} position={'relative'}>
                    {img === undefined ? (
                        <Flex width='100%' height='100%' justifyContent={'center'} alignItems={'center'}>
                            <ProgressCircle isIndeterminate />
                        </Flex>
                    ) : (
                        <img
                            alt={`Camera frame of ${camera.name}`}
                            src={`data:image/jpg;base64,${img}`}
                            style={{
                                objectFit: 'contain',
                                height: '100%',
                                width: '100%',
                            }}
                        />
                    )}
                </View>
                <div className={classes.cameraTag}> {camera.name} </div>
            </Well>
        </Flex>
    );
};
