import { Flex, ProgressCircle } from '@geti-ui/ui';

import { CameraView } from '../../cameras/camera-view';
import { useRuntimeSession } from '../runtime-session-provider';

export const CameraCell = ({ camera_id }: { camera_id: string }) => {
    const { state, statuses, frame } = useRuntimeSession();
    const camera = state?.environment?.cameras.find((candidate) => candidate.camera_id === camera_id);
    const status = camera && statuses[`environment/${camera.key}`];
    // A camera reports ``starting`` with a reason only after it stopped sending frames and is reconnecting.
    const isReconnecting = status?.state === 'starting' && Boolean(status.error);

    if (camera === undefined) {
        return (
            <Flex width='100%' height='100%' justifyContent='center' alignItems='center'>
                <ProgressCircle isIndeterminate aria-label='Connecting camera' />
            </Flex>
        );
    }
    return (
        <CameraView
            source={frame(camera.key)}
            label={`Camera: ${camera.key}`}
            width={camera.width}
            height={camera.height}
            warning={isReconnecting ? 'Camera frozen: no new frames. Reconnecting…' : null}
        />
    );
};
