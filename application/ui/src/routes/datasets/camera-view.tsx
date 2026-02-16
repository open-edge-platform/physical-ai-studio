import { RefObject, useState } from 'react';

import { Flex, ProgressCircle, View, Well } from '@geti/ui';

import { components } from '../../api/openapi-spec';
import { Observation } from './record/use-teleoperation';
import { useInterval } from './use-interval';

import classes from './episode-viewer.module.scss';

type SchemaCamera =
    | components['schemas']['USBCamera-Output']
    | components['schemas']['IPCamera-Output']
    | components['schemas']['BaslerCamera-Output']
    | components['schemas']['RealsenseCamera-Output']
    | components['schemas']['GenicamCamera-Output'];

interface CameraViewProps {
    observation: RefObject<Observation | undefined>;
    camera: SchemaCamera;
}

export const CameraView = ({ camera, observation }: CameraViewProps) => {
    const [img, setImg] = useState<string>();

    useInterval(() => {
        const id = camera.id;
        if (id !== undefined && observation.current?.cameras[id]) {
            setImg(observation.current.cameras[id]);
        }
    }, 1000 / 30); //TODO: Change hardcoding

    const aspectRatio = 640 / 480; //Change hardcoding

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
