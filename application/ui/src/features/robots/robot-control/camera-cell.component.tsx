import { useState } from 'react';

import { Flex, ProgressCircle, View } from '@geti/ui';

import { useInterval } from '../../../routes/datasets/use-interval';
import { useRobotControl } from '../robot-control-provider';

export const CameraCell = ({ camera_id, camera_name }: { camera_id: string; camera_name: string }) => {
    const [img, setImg] = useState<string>();
    const { observation } = useRobotControl();

    useInterval(() => {
        const id = camera_id;
        if (id !== undefined && observation.current?.cameras[id]) {
            setImg(observation.current.cameras[id]);
        }
    }, 1000 / 30); //TODO: Change hardcoding

    const aspectRatio = 640 / 480; //Change hardcoding

    return (
        <Flex UNSAFE_style={{ aspectRatio }}>
            <View height={'100%'} position={'relative'}>
                {img === undefined ? (
                    <Flex width='100%' height='100%' justifyContent={'center'} alignItems={'center'}>
                        <ProgressCircle isIndeterminate />
                    </Flex>
                ) : (
                    <img
                        alt={`Camera frame of ${camera_name}`}
                        src={`data:image/jpg;base64,${img}`}
                        style={{
                            objectFit: 'contain',
                            height: '100%',
                            width: '100%',
                        }}
                    />
                )}
            </View>
        </Flex>
    );
};
