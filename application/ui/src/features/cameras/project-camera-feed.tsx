import { $api } from '../../api/client';
import { SchemaSchemasRobotCamera as SchemaCamera } from '../../api/openapi-spec';
import { CameraFeed } from './camera-feed';

export const ProjectCameraFeed = ({ camera: projectCamera }: { camera: SchemaCamera }) => {
    const { data: hardwareCameras } = $api.useSuspenseQuery('get', '/api/hardware/cameras');
    const hardwareCamera = hardwareCameras.find(({ fingerprint }) => {
        return projectCamera.fingerprint === fingerprint;
    });

    const camera = {
        name: projectCamera.name,
        hardware_name: projectCamera.hardware_name,
        driver: hardwareCamera?.driver ?? projectCamera.driver ?? '',
        fingerprint: hardwareCamera?.fingerprint ?? projectCamera.fingerprint ?? '',
        fps: projectCamera.payload.fps,
        width: projectCamera.payload.width,
        height: projectCamera.payload.height,
        payload: projectCamera.payload,
    };

    return <CameraFeed camera={camera} />;
};
