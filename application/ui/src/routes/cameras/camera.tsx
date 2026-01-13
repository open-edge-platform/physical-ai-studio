import { CameraFeed } from '../../features/cameras/camera-feed';
import { useCamera } from '../../features/robots/use-camera';

export const Camera = () => {
    const projectCamera = useCamera();

    return <CameraFeed camera={projectCamera} />;
};
