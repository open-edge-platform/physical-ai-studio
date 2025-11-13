import { RefObject, useEffect, useRef, useState } from 'react';

import { View } from '@geti/ui';

import { SchemaCameraConfigOutput } from '../../../api/openapi-spec';
import { Observation } from './use-teleoperation';

function useInterval(callback: () => void, delay: number) {
    const savedCallback = useRef<() => void>(callback);

    // Remember the latest callback.
    useEffect(() => {
        savedCallback.current = callback;
    }, [callback]);

    // Set up the interval.
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
export const CameraView = ({ observation, camera }: CameraViewProps) => {
    const [img, setImg] = useState<string>();

    useInterval(() => {
        if (observation.current?.cameras[camera.name]) {
            setImg(observation.current.cameras[camera.name]);
        }
    }, 1000 / camera.fps);

    if (img) {
        return (
            <img
                alt={`Camera frame of ${camera.name}`}
                src={`data:image/jpg;base64,${img}`}
                style={{
                    objectFit: 'contain',
                    aspectRatio: camera.width / camera.height,
                    height: '400px',
                }}
            />
        );
    }
    return <View flex={1} backgroundColor={'gray-400'} width={camera.width} height={camera.height}></View>;
};
