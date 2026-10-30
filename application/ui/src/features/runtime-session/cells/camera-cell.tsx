import { Flex, ProgressCircle } from '@geti-ui/ui';

import { SchemaSessionCamera } from '../../../api/openapi-spec';
import { CameraView } from '../../cameras/camera-view';
import { useRuntimeSession, useSessionCamera } from '../runtime-session-provider';

const SessionCameraView = ({ camera }: { camera: SchemaSessionCamera }) => {
    const source = useSessionCamera(camera.key);
    return <CameraView source={source} label={`Camera: ${camera.key}`} width={camera.width} height={camera.height} />;
};

export const CameraCell = ({ camera_id }: { camera_id: string }) => {
    const { state } = useRuntimeSession();
    const camera = state?.environment?.cameras.find((candidate) => candidate.camera_id === camera_id);

    if (camera === undefined) {
        return (
            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                <ProgressCircle isIndeterminate aria-label='Connecting camera' />
            </Flex>
        );
    }
    return <SessionCameraView camera={camera} />;
};
