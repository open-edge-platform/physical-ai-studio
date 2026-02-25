import { View } from '@geti/ui';

import { CameraFeed } from '../../features/cameras/camera-feed';
import { useCamera } from '../../features/robots/use-camera';

export const Camera = () => {
    const projectCamera = useCamera();

    return (
        <View padding='size-400'>
            <CameraFeed camera={projectCamera} />
        </View>
    );
};
