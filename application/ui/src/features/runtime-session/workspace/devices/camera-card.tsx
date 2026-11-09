import { SchemaSessionCamera, SchemaStatusMessage } from '../../../../api/openapi-spec';
import { SchemaProjectCamera } from '../../../../api/types';
import { CameraIcon } from '../../../robots/camera-form/camera-icon';
import { DeviceCard } from './device-card';
import { getCameraStatus } from './session-devices';

import classes from './devices.module.css';

interface CameraCardProps {
    camera: SchemaSessionCamera;
    projectCamera: SchemaProjectCamera | undefined;
    status: SchemaStatusMessage;
}

export const CameraCard = ({ camera, projectCamera, status }: CameraCardProps) => (
    <DeviceCard
        thumb={
            <div className={classes.cameraIcon}>
                <CameraIcon type={projectCamera?.driver ?? ''} width='100%' />
            </div>
        }
        title={camera.name}
        subtitle={projectCamera ? `${projectCamera.driver} · ${projectCamera.payload.fps} fps` : 'Unknown camera'}
        status={getCameraStatus(status)}
        error={status.error}
    />
);
